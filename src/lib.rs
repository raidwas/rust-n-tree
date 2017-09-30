#![deny(missing_docs)]
#![cfg_attr(test, deny(warnings))]
#![cfg_attr(test, feature(test))]
#![cfg_attr(test, feature(rand))]


//! A generic, n-dimensional quadtree for fast neighbor lookups on multiple axes.

extern crate ref_slice;

use std::{mem, slice};
use self::NTreeVariant::{Branch, Bucket};

#[cfg(test)]
mod test;

/// The required interface for Regions in this n-tree.
///
/// Regions must be able to split themselves, tell if they overlap
/// other regions, and tell if a point is contained within the region.
pub trait Region<P>: Clone {
    /// Does this region contain this point?
    fn contains(&self, &P) -> bool;

    /// Split this region, returning a Vec of sub-regions.
    ///
    /// Invariants:
    ///   - The sub-regions must NOT overlap.
    ///   - All points in self must be contained within one and only one sub-region.
    fn split(&self) -> Vec<Self>;

    /// Does this region overlap with this other region?
    fn overlaps(&self, other: &Self) -> bool;
}

/// A quadtree-like structure, but for arbitrary arity.
///
/// Regions can split themselves into arbitrary numbers of splits,
/// allowing this structure to be used to index data by any number
/// of attributes and quickly query for data that falls within a
/// specific range.
pub struct NTree<R, P> {
    root: NTreeNode<R, P>,
    bucket_limit: u8
}

struct NTreeNode<R, P>{
    region: R,
    kind: NTreeVariant<R, P>,
}

enum NTreeVariant<R, P> {
    /// A leaf of the tree, which contains points.
    Bucket {
        points: Vec<P>,
    },
    /// An interior node of the tree, which contains n subtrees.
    Branch {
        subregions: Vec<NTreeNode<R, P>>
    }
}

impl<P, R: Region<P>> NTree<R, P> {
    /// Create a new n-tree which contains points within
    /// the region and whose buckets are limited to the passed-in size.
    ///
    /// The number of regions returned by region.split() dictates
    /// the arity of the tree.
    pub fn new(region: R, size: u8) -> NTree<R, P> {
        NTree {
            root: NTreeNode{
                region: region,
                kind: Bucket {
                    points: vec![]
                },
            },
            bucket_limit: size
        }
    }

    /// Insert a point into the n-tree, returns true if the point
    /// is within the n-tree and was inserted and false if not.
    pub fn insert(&mut self, point: P) -> bool {
        if !self.root.region.contains(&point) { return false }
        self.root.insert(point, self.bucket_limit)
    }

    /// Get all the points which within the queried region.
    ///
    /// Finds all points which are located in regions overlapping
    /// the passed in region, then filters out all points which
    /// are not strictly within the region.

    pub fn range_query<'t, 'q>(&'t self, query: &'q R) -> RangeQuery<'t, 'q, R, P> {
        RangeQuery {
            query: query,
            points: (&[]).iter(),
            stack: vec![ref_slice::ref_slice(&self.root).iter()],
        }
    }

    /// Is the point contained in the n-tree?
    pub fn contains(&self, point: &P) -> bool {
        self.root.contains(point)
    }


    /// Get all the points nearby a specified point.
    ///
    /// This will return no more than bucket_limit points.
    pub fn nearby<'a>(&'a self, point: &P) -> Option<&'a[P]> {
        if !self.root.region.contains(&point) { return None }
        Some(self.root.nearby(point))
    }
}

impl<P, R: Region<P>> NTreeNode<R, P> {
    /// Insert a point into the n-tree-node, returns true if the point
    /// is within the n-tree and was inserted and false if not.
    fn insert(&mut self, point: P, bucket_limit: u8) -> bool {
        let mut current_node = self;
        loop{
            match {current_node} {
                &mut NTreeNode {region: _, kind: Branch { ref mut subregions }} => {
                    current_node = subregions
                        .iter_mut()
                        .find(|sub_node| sub_node.region.contains(&point))
                        .unwrap(); //does always exist, due to invariant of R.split()
                },
                mut node  => {
                    match node.kind {
                        Bucket {ref mut points} => {
                            if points.len() as u8 != bucket_limit {
                                points.push(point);
                                return true;
                            }
                        },
                        _ => unreachable!()
                    }
                    // Bucket is full
                    split_and_insert(&mut node, point, bucket_limit);
                    return true;
                }
            }
        }
    }
    ///Asumes that the point is contained in the tree.
    ///Therefore it always returns a bucket, and does not need Option in the return type.
    fn nearby<'a>(&'a self, point: &P) -> &'a[P] {
        let mut current_kind = & self.kind;
        loop {
            match {current_kind} {
                & Bucket { ref points } => { return points.as_slice(); },
                & Branch { ref subregions } => {
                    current_kind = & subregions
                        .iter()
                        .find(|r| r.contains(point))
                        .unwrap() //does always exist, due to invariant of R.split()
                        .kind;

                }
            }
        }
    }
    /// Is the point contained in the n-tree?
    fn contains(&self, point: &P) -> bool {
        self.region.contains(point)
    }
}

fn split_and_insert<P, R: Region<P>>(node: &mut NTreeNode<R, P>, point: P, bucket_limit: u8) {
    let old_points;
    match node.kind {
        // Get the old points.
        Bucket { ref mut points } => {
            old_points = mem::replace(points, vec![]);
        },
        Branch { .. } => unreachable!()
    }

    // Replace the bucket with a split branch.
    node.kind = Branch { subregions: node.region
        .split()
        .into_iter()
        .map(|r| NTreeNode {
            region: r,
            kind: Bucket { points: vec![] }
        })
        .collect()
    };

    // Insert all the old points into the right place.
    for old_point in old_points.into_iter() {
        node.insert(old_point, bucket_limit);
    }

    // Finally, insert the new point.
    node.insert(point, bucket_limit);
}

/// An iterator over the points within a region.

// This iterates over the leaves of the tree from left-to-right by
// maintaining (a) the sequence of points at the current level
// (possibly empty), and (b) stack of iterators over the remaining
// children of the parents of the current point.
pub struct RangeQuery<'t,'q, R: 'q + 't, P: 't> {
    query: &'q R,
    points: slice::Iter<'t, P>,
    stack: Vec<slice::Iter<'t, NTreeNode<R, P>>>
}

impl<'t, 'q, R: Region<P>, P> Iterator for RangeQuery<'t, 'q, R, P> {
    type Item = &'t P;

    fn next(&mut self) -> Option<&'t P> {
        'outer: loop {
            // try to find the next point in the region we're
            // currently examining.
            for p in &mut self.points {
                if self.query.contains(p) {
                    return Some(p)
                }
            }

            // no relevant points, so lets find a new region.

            'region_search: loop {
                let mut children_iter = match self.stack.pop() {
                    Some(x) => x,

                    // no more regions, so we're over.
                    None => return None,
                };

                'children: loop {
                    // look at the next item in the current sequence
                    // of children.
                    match children_iter.next() {
                        // this region is empty, next region!
                        None => continue 'region_search,

                        Some(value) => {
                            if value.region.overlaps(self.query) {
                                // we always need to save this state, either we
                                // recur into a new region, or we break out and
                                // handle the points; either way, this is the
                                // last we touch `children_iter` for a little
                                // while.
                                self.stack.push(children_iter);

                                match value.kind {
                                    Bucket { ref points, .. } => {
                                        // found something with points
                                        self.points = points.iter();
                                        continue 'outer;
                                    }
                                    // step down into nested regions.
                                    Branch { ref subregions } => children_iter = subregions.iter()
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
